; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 65523

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child227 - child
    bread224 - bread-portion
    content144 - content-portion
    tray450 tray65 tray201 tray33 - tray
    table21 table309 table155 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray450 kitchen)
     (at tray65 kitchen)
     (at tray201 kitchen)
     (at tray33 kitchen)
     (at_kitchen_bread bread224)
     (at_kitchen_content content144)
     (not_allergic_gluten child227)
     (waiting child227 table309)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child227)
    )
  )
)
