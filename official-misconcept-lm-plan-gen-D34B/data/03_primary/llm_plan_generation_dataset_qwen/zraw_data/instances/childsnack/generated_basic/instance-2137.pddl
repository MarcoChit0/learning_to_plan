; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 373200

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child344 child164 - child
    bread287 bread482 - bread-portion
    content401 content262 - content-portion
    tray283 tray25 tray291 - tray
    table250 table312 table78 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray283 kitchen)
     (at tray25 kitchen)
     (at tray291 kitchen)
     (at_kitchen_bread bread287)
     (at_kitchen_bread bread482)
     (at_kitchen_content content401)
     (at_kitchen_content content262)
     (no_gluten_bread bread287)
     (no_gluten_content content262)
     (allergic_gluten child164)
     (not_allergic_gluten child344)
     (waiting child344 table78)
     (waiting child164 table78)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child344)
     (served child164)
    )
  )
)
