; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 650958

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child170 - child
    bread54 - bread-portion
    content418 - content-portion
    tray207 tray241 tray9 - tray
    table166 table117 table293 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray207 kitchen)
     (at tray241 kitchen)
     (at tray9 kitchen)
     (at_kitchen_bread bread54)
     (at_kitchen_content content418)
     (not_allergic_gluten child170)
     (waiting child170 table166)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child170)
    )
  )
)
