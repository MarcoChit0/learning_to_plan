; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 684284

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child252 - child
    bread448 - bread-portion
    content420 - content-portion
    tray137 tray27 tray256 tray112 - tray
    table267 table131 table42 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray137 kitchen)
     (at tray27 kitchen)
     (at tray256 kitchen)
     (at tray112 kitchen)
     (at_kitchen_bread bread448)
     (at_kitchen_content content420)
     (not_allergic_gluten child252)
     (waiting child252 table42)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child252)
    )
  )
)
