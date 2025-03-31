; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 747802

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child110 - child
    bread229 - bread-portion
    content20 - content-portion
    tray45 tray66 tray233 tray133 - tray
    table458 table59 table154 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray45 kitchen)
     (at tray66 kitchen)
     (at tray233 kitchen)
     (at tray133 kitchen)
     (at_kitchen_bread bread229)
     (at_kitchen_content content20)
     (not_allergic_gluten child110)
     (waiting child110 table154)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child110)
    )
  )
)
