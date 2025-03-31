; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 811628

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child350 - child
    bread353 - bread-portion
    content190 - content-portion
    tray155 - tray
    table239 table159 table111 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray155 kitchen)
     (at_kitchen_bread bread353)
     (at_kitchen_content content190)
     (not_allergic_gluten child350)
     (waiting child350 table159)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child350)
    )
  )
)
