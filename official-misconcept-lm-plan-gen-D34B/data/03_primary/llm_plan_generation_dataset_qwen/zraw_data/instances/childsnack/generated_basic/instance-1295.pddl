; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.5
; random seed: 399909

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child409 child459 - child
    bread218 bread53 - bread-portion
    content375 content205 - content-portion
    tray466 tray52 tray270 tray365 - tray
    table459 table207 table339 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray466 kitchen)
     (at tray52 kitchen)
     (at tray270 kitchen)
     (at tray365 kitchen)
     (at_kitchen_bread bread218)
     (at_kitchen_bread bread53)
     (at_kitchen_content content375)
     (at_kitchen_content content205)
     (no_gluten_bread bread218)
     (no_gluten_content content205)
     (allergic_gluten child409)
     (not_allergic_gluten child459)
     (waiting child409 table459)
     (waiting child459 table459)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child409)
     (served child459)
    )
  )
)
