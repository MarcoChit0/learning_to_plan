; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 753642

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child64 - child
    bread459 - bread-portion
    content18 - content-portion
    tray180 tray407 tray185 - tray
    table110 table102 table14 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray180 kitchen)
     (at tray407 kitchen)
     (at tray185 kitchen)
     (at_kitchen_bread bread459)
     (at_kitchen_content content18)
     (not_allergic_gluten child64)
     (waiting child64 table14)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child64)
    )
  )
)
