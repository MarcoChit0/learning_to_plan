; child-snack task with 2 children and 0.62 gluten factor 
; constant factor of 1.3
; random seed: 582797

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child355 child399 - child
    bread347 bread425 - bread-portion
    content207 content281 - content-portion
    tray126 tray68 tray155 tray14 - tray
    table487 table44 table225 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray126 kitchen)
     (at tray68 kitchen)
     (at tray155 kitchen)
     (at tray14 kitchen)
     (at_kitchen_bread bread347)
     (at_kitchen_bread bread425)
     (at_kitchen_content content207)
     (at_kitchen_content content281)
     (no_gluten_bread bread425)
     (no_gluten_content content207)
     (allergic_gluten child355)
     (not_allergic_gluten child399)
     (waiting child355 table225)
     (waiting child399 table44)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child355)
     (served child399)
    )
  )
)
