; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 482731

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child110 - child
    bread370 - bread-portion
    content41 - content-portion
    tray351 tray102 tray66 tray295 - tray
    table289 table41 table184 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray351 kitchen)
     (at tray102 kitchen)
     (at tray66 kitchen)
     (at tray295 kitchen)
     (at_kitchen_bread bread370)
     (at_kitchen_content content41)
     (not_allergic_gluten child110)
     (waiting child110 table41)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child110)
    )
  )
)
