; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 583276

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child470 - child
    bread15 - bread-portion
    content55 - content-portion
    tray223 - tray
    table298 table112 table317 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray223 kitchen)
     (at_kitchen_bread bread15)
     (at_kitchen_content content55)
     (not_allergic_gluten child470)
     (waiting child470 table317)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child470)
    )
  )
)
