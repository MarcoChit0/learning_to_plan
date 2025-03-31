; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 81361

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child81 child176 - child
    bread91 bread162 - bread-portion
    content290 content2 - content-portion
    tray203 tray389 - tray
    table57 table349 table364 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray203 kitchen)
     (at tray389 kitchen)
     (at_kitchen_bread bread91)
     (at_kitchen_bread bread162)
     (at_kitchen_content content290)
     (at_kitchen_content content2)
     (not_allergic_gluten child176)
     (not_allergic_gluten child81)
     (waiting child81 table57)
     (waiting child176 table349)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child81)
     (served child176)
    )
  )
)
