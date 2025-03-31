; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 278386

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child5 child246 - child
    bread66 bread82 - bread-portion
    content46 content162 - content-portion
    tray222 tray144 tray352 - tray
    table214 table379 table433 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray222 kitchen)
     (at tray144 kitchen)
     (at tray352 kitchen)
     (at_kitchen_bread bread66)
     (at_kitchen_bread bread82)
     (at_kitchen_content content46)
     (at_kitchen_content content162)
     (not_allergic_gluten child5)
     (not_allergic_gluten child246)
     (waiting child5 table433)
     (waiting child246 table379)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child5)
     (served child246)
    )
  )
)
