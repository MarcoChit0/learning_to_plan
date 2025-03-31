; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 975385

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child64 child486 - child
    bread129 bread270 - bread-portion
    content46 content139 - content-portion
    tray45 tray37 tray271 - tray
    table446 table308 table105 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray45 kitchen)
     (at tray37 kitchen)
     (at tray271 kitchen)
     (at_kitchen_bread bread129)
     (at_kitchen_bread bread270)
     (at_kitchen_content content46)
     (at_kitchen_content content139)
     (no_gluten_bread bread129)
     (no_gluten_content content46)
     (allergic_gluten child64)
     (not_allergic_gluten child486)
     (waiting child64 table446)
     (waiting child486 table446)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child64)
     (served child486)
    )
  )
)
