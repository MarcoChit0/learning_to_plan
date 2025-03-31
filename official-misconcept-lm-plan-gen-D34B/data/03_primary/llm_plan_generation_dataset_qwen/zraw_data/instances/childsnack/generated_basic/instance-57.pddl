; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 183706

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child55 child249 - child
    bread158 bread326 - bread-portion
    content493 content13 - content-portion
    tray326 tray334 - tray
    table55 table207 table261 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray326 kitchen)
     (at tray334 kitchen)
     (at_kitchen_bread bread158)
     (at_kitchen_bread bread326)
     (at_kitchen_content content493)
     (at_kitchen_content content13)
     (not_allergic_gluten child249)
     (not_allergic_gluten child55)
     (waiting child55 table261)
     (waiting child249 table55)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child55)
     (served child249)
    )
  )
)
