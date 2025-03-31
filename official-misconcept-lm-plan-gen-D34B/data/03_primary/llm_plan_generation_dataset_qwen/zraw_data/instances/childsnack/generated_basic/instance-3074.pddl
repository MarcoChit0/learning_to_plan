; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 735455

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child494 child438 - child
    bread322 bread148 - bread-portion
    content431 content244 - content-portion
    tray287 tray155 tray271 - tray
    table135 table139 table198 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray287 kitchen)
     (at tray155 kitchen)
     (at tray271 kitchen)
     (at_kitchen_bread bread322)
     (at_kitchen_bread bread148)
     (at_kitchen_content content431)
     (at_kitchen_content content244)
     (no_gluten_bread bread322)
     (no_gluten_content content431)
     (allergic_gluten child438)
     (not_allergic_gluten child494)
     (waiting child494 table198)
     (waiting child438 table139)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child494)
     (served child438)
    )
  )
)
