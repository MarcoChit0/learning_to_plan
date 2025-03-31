; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 663391

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child12 - child
    bread447 - bread-portion
    content133 - content-portion
    tray292 tray210 tray35 tray45 - tray
    table240 table59 table86 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray292 kitchen)
     (at tray210 kitchen)
     (at tray35 kitchen)
     (at tray45 kitchen)
     (at_kitchen_bread bread447)
     (at_kitchen_content content133)
     (not_allergic_gluten child12)
     (waiting child12 table86)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child12)
    )
  )
)
