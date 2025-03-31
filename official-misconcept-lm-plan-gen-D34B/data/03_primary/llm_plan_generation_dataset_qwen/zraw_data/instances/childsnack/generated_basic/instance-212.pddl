; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 227155

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child486 child476 - child
    bread252 bread358 - bread-portion
    content405 content300 - content-portion
    tray74 tray146 tray331 - tray
    table172 table388 table430 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray74 kitchen)
     (at tray146 kitchen)
     (at tray331 kitchen)
     (at_kitchen_bread bread252)
     (at_kitchen_bread bread358)
     (at_kitchen_content content405)
     (at_kitchen_content content300)
     (not_allergic_gluten child476)
     (not_allergic_gluten child486)
     (waiting child486 table388)
     (waiting child476 table430)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child486)
     (served child476)
    )
  )
)
