; child-snack task with 3 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 748000

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child252 child369 child166 - child
    bread238 bread368 bread178 - bread-portion
    content178 content221 content120 - content-portion
    tray85 - tray
    table211 table199 table119 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray85 kitchen)
     (at_kitchen_bread bread238)
     (at_kitchen_bread bread368)
     (at_kitchen_bread bread178)
     (at_kitchen_content content178)
     (at_kitchen_content content221)
     (at_kitchen_content content120)
     (no_gluten_bread bread238)
     (no_gluten_bread bread368)
     (no_gluten_content content178)
     (no_gluten_content content221)
     (allergic_gluten child166)
     (allergic_gluten child252)
     (not_allergic_gluten child369)
     (waiting child252 table119)
     (waiting child369 table199)
     (waiting child166 table199)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child252)
     (served child369)
     (served child166)
    )
  )
)
