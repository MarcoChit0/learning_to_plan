; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 929048

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child301 child75 - child
    bread308 bread266 - bread-portion
    content167 content93 - content-portion
    tray193 tray391 tray475 - tray
    table69 table292 table70 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray193 kitchen)
     (at tray391 kitchen)
     (at tray475 kitchen)
     (at_kitchen_bread bread308)
     (at_kitchen_bread bread266)
     (at_kitchen_content content167)
     (at_kitchen_content content93)
     (not_allergic_gluten child301)
     (not_allergic_gluten child75)
     (waiting child301 table70)
     (waiting child75 table292)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child301)
     (served child75)
    )
  )
)
