; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 14282

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child59 - child
    bread266 - bread-portion
    content475 - content-portion
    tray3 tray317 tray188 tray223 - tray
    table321 table162 table424 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray3 kitchen)
     (at tray317 kitchen)
     (at tray188 kitchen)
     (at tray223 kitchen)
     (at_kitchen_bread bread266)
     (at_kitchen_content content475)
     (not_allergic_gluten child59)
     (waiting child59 table424)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child59)
    )
  )
)
