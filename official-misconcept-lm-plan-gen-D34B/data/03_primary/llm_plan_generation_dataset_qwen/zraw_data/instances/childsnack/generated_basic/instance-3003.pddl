; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 503106

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child394 child336 - child
    bread417 bread439 - bread-portion
    content151 content134 - content-portion
    tray326 tray260 tray155 tray474 - tray
    table300 table481 table295 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray326 kitchen)
     (at tray260 kitchen)
     (at tray155 kitchen)
     (at tray474 kitchen)
     (at_kitchen_bread bread417)
     (at_kitchen_bread bread439)
     (at_kitchen_content content151)
     (at_kitchen_content content134)
     (not_allergic_gluten child394)
     (not_allergic_gluten child336)
     (waiting child394 table481)
     (waiting child336 table295)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child394)
     (served child336)
    )
  )
)
