; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 685861

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child192 - child
    bread213 - bread-portion
    content323 - content-portion
    tray312 tray412 - tray
    table324 table260 table8 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray312 kitchen)
     (at tray412 kitchen)
     (at_kitchen_bread bread213)
     (at_kitchen_content content323)
     (not_allergic_gluten child192)
     (waiting child192 table8)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child192)
    )
  )
)
