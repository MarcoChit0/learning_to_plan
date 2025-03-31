; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 773039

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child168 child78 - child
    bread221 bread73 - bread-portion
    content99 content200 - content-portion
    tray255 tray328 - tray
    table21 table410 table233 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray255 kitchen)
     (at tray328 kitchen)
     (at_kitchen_bread bread221)
     (at_kitchen_bread bread73)
     (at_kitchen_content content99)
     (at_kitchen_content content200)
     (no_gluten_bread bread73)
     (no_gluten_content content99)
     (allergic_gluten child78)
     (not_allergic_gluten child168)
     (waiting child168 table21)
     (waiting child78 table233)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child168)
     (served child78)
    )
  )
)
