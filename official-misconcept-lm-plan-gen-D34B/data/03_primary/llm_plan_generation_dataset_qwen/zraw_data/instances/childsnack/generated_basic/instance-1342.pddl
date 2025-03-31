; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 659247

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child333 child250 - child
    bread422 bread369 - bread-portion
    content169 content471 - content-portion
    tray139 tray114 tray41 - tray
    table298 table208 table454 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray139 kitchen)
     (at tray114 kitchen)
     (at tray41 kitchen)
     (at_kitchen_bread bread422)
     (at_kitchen_bread bread369)
     (at_kitchen_content content169)
     (at_kitchen_content content471)
     (not_allergic_gluten child333)
     (not_allergic_gluten child250)
     (waiting child333 table454)
     (waiting child250 table454)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child333)
     (served child250)
    )
  )
)
