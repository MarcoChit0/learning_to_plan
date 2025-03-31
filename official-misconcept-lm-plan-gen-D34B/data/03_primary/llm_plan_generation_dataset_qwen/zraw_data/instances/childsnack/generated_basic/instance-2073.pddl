; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 240429

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child76 - child
    bread86 - bread-portion
    content203 - content-portion
    tray55 tray272 tray195 tray425 - tray
    table180 table321 table158 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray55 kitchen)
     (at tray272 kitchen)
     (at tray195 kitchen)
     (at tray425 kitchen)
     (at_kitchen_bread bread86)
     (at_kitchen_content content203)
     (not_allergic_gluten child76)
     (waiting child76 table180)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child76)
    )
  )
)
