; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 945510

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child198 child473 - child
    bread351 bread60 - bread-portion
    content448 content438 - content-portion
    tray480 tray493 - tray
    table386 table118 table73 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray480 kitchen)
     (at tray493 kitchen)
     (at_kitchen_bread bread351)
     (at_kitchen_bread bread60)
     (at_kitchen_content content448)
     (at_kitchen_content content438)
     (not_allergic_gluten child473)
     (not_allergic_gluten child198)
     (waiting child198 table386)
     (waiting child473 table118)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child198)
     (served child473)
    )
  )
)
