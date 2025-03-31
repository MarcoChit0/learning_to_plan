; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 255487

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child284 child367 child56 - child
    bread495 bread496 bread429 - bread-portion
    content400 content379 content294 - content-portion
    tray410 - tray
    table106 table225 table253 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray410 kitchen)
     (at_kitchen_bread bread495)
     (at_kitchen_bread bread496)
     (at_kitchen_bread bread429)
     (at_kitchen_content content400)
     (at_kitchen_content content379)
     (at_kitchen_content content294)
     (not_allergic_gluten child56)
     (not_allergic_gluten child367)
     (not_allergic_gluten child284)
     (waiting child284 table253)
     (waiting child367 table253)
     (waiting child56 table253)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child284)
     (served child367)
     (served child56)
    )
  )
)
