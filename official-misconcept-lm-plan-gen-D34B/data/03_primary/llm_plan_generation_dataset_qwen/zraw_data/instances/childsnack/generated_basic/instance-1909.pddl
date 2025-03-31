; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 240094

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child9 - child
    bread397 - bread-portion
    content205 - content-portion
    tray290 tray123 tray209 - tray
    table311 table43 table236 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray290 kitchen)
     (at tray123 kitchen)
     (at tray209 kitchen)
     (at_kitchen_bread bread397)
     (at_kitchen_content content205)
     (not_allergic_gluten child9)
     (waiting child9 table236)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child9)
    )
  )
)
