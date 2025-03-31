; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 465268

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child164 - child
    bread381 - bread-portion
    content395 - content-portion
    tray151 tray246 tray188 tray343 - tray
    table16 table497 table376 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray151 kitchen)
     (at tray246 kitchen)
     (at tray188 kitchen)
     (at tray343 kitchen)
     (at_kitchen_bread bread381)
     (at_kitchen_content content395)
     (not_allergic_gluten child164)
     (waiting child164 table376)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child164)
    )
  )
)
