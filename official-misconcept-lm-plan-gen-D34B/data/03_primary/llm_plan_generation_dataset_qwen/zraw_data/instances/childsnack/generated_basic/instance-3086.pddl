; child-snack task with 2 children and 0.45 gluten factor 
; constant factor of 1.7
; random seed: 698166

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child175 child384 - child
    bread57 bread269 - bread-portion
    content357 content86 - content-portion
    tray217 tray137 tray402 - tray
    table427 table30 table204 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray217 kitchen)
     (at tray137 kitchen)
     (at tray402 kitchen)
     (at_kitchen_bread bread57)
     (at_kitchen_bread bread269)
     (at_kitchen_content content357)
     (at_kitchen_content content86)
     (not_allergic_gluten child175)
     (not_allergic_gluten child384)
     (waiting child175 table204)
     (waiting child384 table204)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child175)
     (served child384)
    )
  )
)
