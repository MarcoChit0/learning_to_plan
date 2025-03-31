; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 915463

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child58 - child
    bread11 - bread-portion
    content227 - content-portion
    tray495 tray262 tray9 - tray
    table185 table118 table382 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray495 kitchen)
     (at tray262 kitchen)
     (at tray9 kitchen)
     (at_kitchen_bread bread11)
     (at_kitchen_content content227)
     (not_allergic_gluten child58)
     (waiting child58 table118)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child58)
    )
  )
)
