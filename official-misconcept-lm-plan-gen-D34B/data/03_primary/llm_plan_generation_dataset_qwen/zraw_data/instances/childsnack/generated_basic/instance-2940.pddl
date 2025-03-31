; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 306980

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child319 - child
    bread174 - bread-portion
    content174 - content-portion
    tray164 tray207 tray326 tray250 - tray
    table344 table476 table212 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray164 kitchen)
     (at tray207 kitchen)
     (at tray326 kitchen)
     (at tray250 kitchen)
     (at_kitchen_bread bread174)
     (at_kitchen_content content174)
     (not_allergic_gluten child319)
     (waiting child319 table212)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child319)
    )
  )
)
