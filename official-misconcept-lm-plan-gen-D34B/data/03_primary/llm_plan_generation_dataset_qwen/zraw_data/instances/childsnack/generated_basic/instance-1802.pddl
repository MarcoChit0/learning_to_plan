; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 753975

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child288 - child
    bread264 - bread-portion
    content340 - content-portion
    tray332 - tray
    table335 table21 table326 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray332 kitchen)
     (at_kitchen_bread bread264)
     (at_kitchen_content content340)
     (not_allergic_gluten child288)
     (waiting child288 table21)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child288)
    )
  )
)
