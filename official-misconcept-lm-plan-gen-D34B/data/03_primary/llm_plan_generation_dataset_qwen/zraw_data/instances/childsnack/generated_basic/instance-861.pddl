; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 580966

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child11 child179 - child
    bread106 bread322 - bread-portion
    content406 content4 - content-portion
    tray260 tray53 - tray
    table327 table197 table364 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray260 kitchen)
     (at tray53 kitchen)
     (at_kitchen_bread bread106)
     (at_kitchen_bread bread322)
     (at_kitchen_content content406)
     (at_kitchen_content content4)
     (no_gluten_bread bread106)
     (no_gluten_content content4)
     (allergic_gluten child179)
     (not_allergic_gluten child11)
     (waiting child11 table197)
     (waiting child179 table327)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child11)
     (served child179)
    )
  )
)
