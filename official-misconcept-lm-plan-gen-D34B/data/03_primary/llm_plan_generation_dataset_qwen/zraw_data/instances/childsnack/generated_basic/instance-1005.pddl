; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 809623

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child78 child403 child187 - child
    bread321 bread400 bread106 - bread-portion
    content41 content492 content93 - content-portion
    tray113 - tray
    table141 table219 table305 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray113 kitchen)
     (at_kitchen_bread bread321)
     (at_kitchen_bread bread400)
     (at_kitchen_bread bread106)
     (at_kitchen_content content41)
     (at_kitchen_content content492)
     (at_kitchen_content content93)
     (not_allergic_gluten child78)
     (not_allergic_gluten child403)
     (not_allergic_gluten child187)
     (waiting child78 table141)
     (waiting child403 table219)
     (waiting child187 table305)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child78)
     (served child403)
     (served child187)
    )
  )
)
