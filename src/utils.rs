pub fn diff_ids(a: &Vec<u32>, b: &Vec<u32>) -> Vec<u32> {
    let mut result = Vec::new();
    let mut a_idx = 0;
    let mut b_idx = 0;

    while a_idx < a.len() && b_idx < b.len() {
        if a[a_idx] == b[b_idx] {
            a_idx += 1; // Skip common elements
            b_idx += 1;
        } else if a[a_idx] < b[b_idx] {
            // Elements present only in a
            result.push(a[a_idx]);
            a_idx += 1;
        } else {
            // Ignore elements that exist only in b
            b_idx += 1;
        }
    }

    // Add the remaining elements of a (since they do not exist in b)
    while a_idx < a.len() {
        result.push(a[a_idx]);
        a_idx += 1;
    }

    result
}

pub fn intersect_ids(a: &Vec<u32>, b: &Vec<u32>) -> Vec<u32> {
    let mut result = Vec::new();
    let mut a_idx = 0;
    let mut b_idx = 0;

    while a_idx < a.len() && b_idx < b.len() {
        if a[a_idx] == b[b_idx] {
            result.push(a[a_idx]);
            a_idx += 1;
            b_idx += 1;
        } else if a[a_idx] < b[b_idx] {
            a_idx += 1;
        } else {
            b_idx += 1;
        }
    }

    result
}

pub fn sort_list_by_dist(list: &mut Vec<(f32, u32, bool)>) {
    list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Less));
}

pub fn sort_list_by_dist_v1(list: &mut Vec<(f32, u32)>) {
    list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Less));
}

pub fn is_contained_in(i: &u32, vec: &Vec<(f32, u32)>) -> bool {
    !vec.iter()
        .filter(|(_, id)| *id == *i)
        .collect::<Vec<&(f32, u32)>>()
        .is_empty()
}

pub fn insert_id(value: u32, vec: &mut Vec<u32>) {
    match vec.binary_search(&value) {
        Ok(_index) => { // If already exsits
        }
        Err(index) => {
            vec.insert(index, value);
        }
    }
}

pub fn insert_dist(value: (f32, u32), vec: &mut Vec<(f32, u32)>) {
    match vec.binary_search_by(|probe| {
        probe
            .0
            .partial_cmp(&value.0)
            .unwrap_or(std::cmp::Ordering::Less)
    }) {
        Ok(index) => {
            // identify a range of groups of elements with the same f32 value
            let mut start = index;
            while start > 0 && vec[start - 1].0 == value.0 {
                start -= 1;
            }
            let mut end = index;
            while end < vec.len() - 1 && vec[end + 1].0 == value.0 {
                end += 1;
            }

            // Check for elements with the same usize value within the specified range
            if !(start..=end).any(|i| vec[i].1 == value.1) {
                vec.insert(index, value);
            }
        }
        Err(index) => {
            vec.insert(index, value);
        }
    };
}
