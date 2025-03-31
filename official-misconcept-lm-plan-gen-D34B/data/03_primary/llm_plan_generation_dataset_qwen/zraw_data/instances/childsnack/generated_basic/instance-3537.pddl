; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 457938

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child416 child249 - child
    bread377 bread188 - bread-portion
    content289 content263 - content-portion
    tray195 tray340 - tray
    table437 table362 table268 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray195 kitchen)
     (at tray340 kitchen)
     (at_kitchen_bread bread377)
     (at_kitchen_bread bread188)
     (at_kitchen_content content289)
     (at_kitchen_content content263)
     (not_allergic_gluten child249)
     (not_allergic_gluten child416)
     (waiting child416 table268)
     (waiting child249 table437)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child416)
     (served child249)
    )
  )
)
