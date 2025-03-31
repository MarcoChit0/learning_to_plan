; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 767218

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child416 child342 - child
    bread231 bread218 - bread-portion
    content226 content348 - content-portion
    tray65 tray221 tray81 - tray
    table425 table480 table200 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray65 kitchen)
     (at tray221 kitchen)
     (at tray81 kitchen)
     (at_kitchen_bread bread231)
     (at_kitchen_bread bread218)
     (at_kitchen_content content226)
     (at_kitchen_content content348)
     (not_allergic_gluten child416)
     (not_allergic_gluten child342)
     (waiting child416 table480)
     (waiting child342 table480)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child416)
     (served child342)
    )
  )
)
