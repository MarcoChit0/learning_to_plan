; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 297051

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child31 child11 - child
    bread328 bread75 - bread-portion
    content100 content347 - content-portion
    tray40 tray419 - tray
    table53 table64 table469 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray40 kitchen)
     (at tray419 kitchen)
     (at_kitchen_bread bread328)
     (at_kitchen_bread bread75)
     (at_kitchen_content content100)
     (at_kitchen_content content347)
     (no_gluten_bread bread75)
     (no_gluten_content content100)
     (allergic_gluten child11)
     (not_allergic_gluten child31)
     (waiting child31 table469)
     (waiting child11 table53)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child31)
     (served child11)
    )
  )
)
