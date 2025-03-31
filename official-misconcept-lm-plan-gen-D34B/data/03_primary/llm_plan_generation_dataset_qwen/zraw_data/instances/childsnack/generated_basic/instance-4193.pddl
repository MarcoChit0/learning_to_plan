; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 998359

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child438 child346 - child
    bread13 bread88 - bread-portion
    content403 content141 - content-portion
    tray465 - tray
    table379 table424 table324 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray465 kitchen)
     (at_kitchen_bread bread13)
     (at_kitchen_bread bread88)
     (at_kitchen_content content403)
     (at_kitchen_content content141)
     (not_allergic_gluten child346)
     (not_allergic_gluten child438)
     (waiting child438 table424)
     (waiting child346 table379)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child438)
     (served child346)
    )
  )
)
