; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 19316

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child3 child495 - child
    bread5 bread257 - bread-portion
    content288 content317 - content-portion
    tray133 tray92 tray252 - tray
    table70 table495 table484 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray133 kitchen)
     (at tray92 kitchen)
     (at tray252 kitchen)
     (at_kitchen_bread bread5)
     (at_kitchen_bread bread257)
     (at_kitchen_content content288)
     (at_kitchen_content content317)
     (no_gluten_bread bread5)
     (no_gluten_content content317)
     (allergic_gluten child495)
     (not_allergic_gluten child3)
     (waiting child3 table495)
     (waiting child495 table70)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child3)
     (served child495)
    )
  )
)
