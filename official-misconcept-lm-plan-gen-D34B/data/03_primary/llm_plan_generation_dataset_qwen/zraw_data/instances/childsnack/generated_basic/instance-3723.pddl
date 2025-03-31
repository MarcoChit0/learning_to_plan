; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 922269

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child204 - child
    bread55 - bread-portion
    content2 - content-portion
    tray314 tray294 tray300 - tray
    table81 table153 table423 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray314 kitchen)
     (at tray294 kitchen)
     (at tray300 kitchen)
     (at_kitchen_bread bread55)
     (at_kitchen_content content2)
     (not_allergic_gluten child204)
     (waiting child204 table423)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child204)
    )
  )
)
