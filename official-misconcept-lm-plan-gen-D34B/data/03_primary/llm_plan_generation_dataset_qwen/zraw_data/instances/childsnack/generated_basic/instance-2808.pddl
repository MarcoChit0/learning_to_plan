; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 80075

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child76 child316 - child
    bread18 bread455 - bread-portion
    content122 content360 - content-portion
    tray400 tray462 - tray
    table392 table142 table430 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray400 kitchen)
     (at tray462 kitchen)
     (at_kitchen_bread bread18)
     (at_kitchen_bread bread455)
     (at_kitchen_content content122)
     (at_kitchen_content content360)
     (not_allergic_gluten child316)
     (not_allergic_gluten child76)
     (waiting child76 table430)
     (waiting child316 table430)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child76)
     (served child316)
    )
  )
)
