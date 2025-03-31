; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 805869

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child259 child275 - child
    bread46 bread86 - bread-portion
    content125 content58 - content-portion
    tray92 tray394 tray260 tray93 - tray
    table276 table45 table145 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray92 kitchen)
     (at tray394 kitchen)
     (at tray260 kitchen)
     (at tray93 kitchen)
     (at_kitchen_bread bread46)
     (at_kitchen_bread bread86)
     (at_kitchen_content content125)
     (at_kitchen_content content58)
     (not_allergic_gluten child275)
     (not_allergic_gluten child259)
     (waiting child259 table45)
     (waiting child275 table145)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child259)
     (served child275)
    )
  )
)
