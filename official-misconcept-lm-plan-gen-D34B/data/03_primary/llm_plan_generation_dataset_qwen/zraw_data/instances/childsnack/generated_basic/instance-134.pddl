; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 782130

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child158 child439 - child
    bread126 bread230 - bread-portion
    content88 content91 - content-portion
    tray194 tray407 tray49 tray128 - tray
    table298 table303 table79 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray194 kitchen)
     (at tray407 kitchen)
     (at tray49 kitchen)
     (at tray128 kitchen)
     (at_kitchen_bread bread126)
     (at_kitchen_bread bread230)
     (at_kitchen_content content88)
     (at_kitchen_content content91)
     (not_allergic_gluten child439)
     (not_allergic_gluten child158)
     (waiting child158 table303)
     (waiting child439 table298)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child158)
     (served child439)
    )
  )
)
