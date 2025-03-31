; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 494250

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child495 child278 - child
    bread436 bread278 - bread-portion
    content476 content0 - content-portion
    tray267 tray74 tray376 - tray
    table284 table117 table294 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray267 kitchen)
     (at tray74 kitchen)
     (at tray376 kitchen)
     (at_kitchen_bread bread436)
     (at_kitchen_bread bread278)
     (at_kitchen_content content476)
     (at_kitchen_content content0)
     (not_allergic_gluten child278)
     (not_allergic_gluten child495)
     (waiting child495 table284)
     (waiting child278 table117)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child495)
     (served child278)
    )
  )
)
