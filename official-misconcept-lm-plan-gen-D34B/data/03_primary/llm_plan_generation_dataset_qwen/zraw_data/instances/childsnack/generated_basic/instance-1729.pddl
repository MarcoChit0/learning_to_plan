; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 783451

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child349 - child
    bread117 - bread-portion
    content239 - content-portion
    tray197 tray388 tray76 - tray
    table114 table12 table179 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray197 kitchen)
     (at tray388 kitchen)
     (at tray76 kitchen)
     (at_kitchen_bread bread117)
     (at_kitchen_content content239)
     (not_allergic_gluten child349)
     (waiting child349 table179)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child349)
    )
  )
)
