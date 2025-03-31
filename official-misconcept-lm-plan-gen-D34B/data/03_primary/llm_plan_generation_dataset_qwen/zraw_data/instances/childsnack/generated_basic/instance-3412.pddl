; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 513117

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child315 - child
    bread17 - bread-portion
    content196 - content-portion
    tray88 tray53 tray153 tray13 - tray
    table280 table406 table72 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray88 kitchen)
     (at tray53 kitchen)
     (at tray153 kitchen)
     (at tray13 kitchen)
     (at_kitchen_bread bread17)
     (at_kitchen_content content196)
     (not_allergic_gluten child315)
     (waiting child315 table280)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child315)
    )
  )
)
