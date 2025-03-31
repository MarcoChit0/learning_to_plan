; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 39549

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child99 child121 child171 - child
    bread383 bread303 bread435 - bread-portion
    content428 content265 content121 - content-portion
    tray458 - tray
    table370 table107 table11 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray458 kitchen)
     (at_kitchen_bread bread383)
     (at_kitchen_bread bread303)
     (at_kitchen_bread bread435)
     (at_kitchen_content content428)
     (at_kitchen_content content265)
     (at_kitchen_content content121)
     (not_allergic_gluten child99)
     (not_allergic_gluten child171)
     (not_allergic_gluten child121)
     (waiting child99 table370)
     (waiting child121 table107)
     (waiting child171 table11)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child99)
     (served child121)
     (served child171)
    )
  )
)
