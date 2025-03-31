; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 914985

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child19 child330 - child
    bread192 bread117 - bread-portion
    content436 content389 - content-portion
    tray353 tray365 tray471 tray143 - tray
    table259 table347 table97 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray353 kitchen)
     (at tray365 kitchen)
     (at tray471 kitchen)
     (at tray143 kitchen)
     (at_kitchen_bread bread192)
     (at_kitchen_bread bread117)
     (at_kitchen_content content436)
     (at_kitchen_content content389)
     (not_allergic_gluten child330)
     (not_allergic_gluten child19)
     (waiting child19 table347)
     (waiting child330 table259)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child19)
     (served child330)
    )
  )
)
