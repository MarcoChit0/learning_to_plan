; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 491624

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child323 - child
    bread240 - bread-portion
    content360 - content-portion
    tray146 tray399 tray48 - tray
    table366 table65 table228 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray146 kitchen)
     (at tray399 kitchen)
     (at tray48 kitchen)
     (at_kitchen_bread bread240)
     (at_kitchen_content content360)
     (not_allergic_gluten child323)
     (waiting child323 table228)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child323)
    )
  )
)
