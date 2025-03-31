; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 422252

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child151 child18 - child
    bread146 bread257 - bread-portion
    content336 content459 - content-portion
    tray19 tray494 - tray
    table132 table263 table56 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray19 kitchen)
     (at tray494 kitchen)
     (at_kitchen_bread bread146)
     (at_kitchen_bread bread257)
     (at_kitchen_content content336)
     (at_kitchen_content content459)
     (not_allergic_gluten child151)
     (not_allergic_gluten child18)
     (waiting child151 table132)
     (waiting child18 table263)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child151)
     (served child18)
    )
  )
)
