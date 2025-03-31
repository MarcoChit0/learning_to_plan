; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 478706

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child303 child308 - child
    bread360 bread41 - bread-portion
    content433 content438 - content-portion
    tray88 - tray
    table265 table170 table350 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray88 kitchen)
     (at_kitchen_bread bread360)
     (at_kitchen_bread bread41)
     (at_kitchen_content content433)
     (at_kitchen_content content438)
     (not_allergic_gluten child303)
     (not_allergic_gluten child308)
     (waiting child303 table350)
     (waiting child308 table350)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child303)
     (served child308)
    )
  )
)
