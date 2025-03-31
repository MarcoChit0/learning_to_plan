; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 85944

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child312 - child
    bread389 - bread-portion
    content359 - content-portion
    tray221 tray350 tray267 - tray
    table318 table481 table321 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray221 kitchen)
     (at tray350 kitchen)
     (at tray267 kitchen)
     (at_kitchen_bread bread389)
     (at_kitchen_content content359)
     (not_allergic_gluten child312)
     (waiting child312 table321)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child312)
    )
  )
)
